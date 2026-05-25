"""
methodology_trace.py
====================
Lightweight audit trail for every automated statistical decision made during
an analysis run.  Instances are passed through the analysis pipeline and later
rendered into the "Methodology Log" Excel sheet.
"""


class MethodologyTrace:
    """Collect human-readable decision steps produced during statistical analysis.

    Usage
    -----
    trace = MethodologyTrace()
    trace.add(1, "Normality", "Shapiro-Wilk on 'Group A' yielded p=0.031 — normality violated.",
              detail="W=0.891, p=0.031")
    trace.add(2, "Test Selection",
              "Normality violated → switched from One-Way ANOVA to Kruskal-Wallis.")
    log_sheet_writer(workbook, results, fmt, trace=trace)
    """

    # Category → display order (lower = earlier in table)
    _CATEGORY_ORDER = {
        "Data Check":   1,
        "Normality":    2,
        "Assumption":   3,
        "Test Selection": 4,
        "Post-hoc":     5,
        "Correction":   6,
        "Other":        99,
    }

    def __init__(self):
        self.steps: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, step_number: int, category: str, decision: str, detail: str | None = None):
        """Append a decision step.

        Parameters
        ----------
        step_number : int
            Sequential step number (1-based).  Determines row order in the
            Excel table.
        category : str
            High-level category, e.g. "Normality", "Test Selection", "Post-hoc".
        decision : str
            Human-readable sentence describing the automated decision.
        detail : str | None
            Optional supplementary data (e.g. "W=0.891, p=0.031").
        """
        self.steps.append({
            "step":     step_number,
            "category": category,
            "decision": decision,
            "detail":   detail or "",
        })

    def to_list(self) -> list[dict]:
        """Return all steps sorted by step number."""
        return sorted(self.steps, key=lambda s: s["step"])

    def to_methods_paragraph(self) -> str:
        """Generate a flowing prose paragraph suitable for a manuscript Methods section.

        The paragraph is assembled from the collected steps in chronological order.
        """
        if not self.steps:
            return (
                "Statistical analysis was performed using BioMedStatX 2.0. "
                "The significance threshold was set at \u03b1\u202f=\u202f0.05."
            )

        lines = [
            "Statistical analysis was performed using BioMedStatX 2.0."
        ]

        # Gather decisions by category for natural prose flow
        by_cat: dict[str, list[str]] = {}
        for step in self.to_list():
            cat = step["category"]
            by_cat.setdefault(cat, []).append(step["decision"])

        # Normality
        if "Normality" in by_cat:
            lines.append(
                "Normality was assessed using the Shapiro-Wilk test. "
                + " ".join(by_cat["Normality"])
            )

        # Assumption checks (variance, slopes, etc.)
        if "Assumption" in by_cat:
            lines.append(" ".join(by_cat["Assumption"]))

        # Test selection
        if "Test Selection" in by_cat:
            lines.append(" ".join(by_cat["Test Selection"]))

        # Post-hoc
        if "Post-hoc" in by_cat:
            lines.append(" ".join(by_cat["Post-hoc"]))

        # Corrections
        if "Correction" in by_cat:
            lines.append(" ".join(by_cat["Correction"]))

        # Other categories
        for cat, decisions in by_cat.items():
            if cat not in ("Normality", "Assumption", "Test Selection", "Post-hoc", "Correction"):
                lines.append(" ".join(decisions))

        lines.append("The significance threshold was set at \u03b1\u202f=\u202f0.05.")
        return " ".join(lines)
