"""Human-friendly, deterministic ordering of categorical factor levels.

The order depends ONLY on the set of labels, never on DataFrame row order:

    key(label) = (hierarchy_rank(label), natural_key(label))

- ``natural_key`` is numeric-aware, so ``6h < 24h < 48h`` sort by value no matter
  what text surrounds the number (fixes lexicographic ``24h, 48h, 6h``).
- ``hierarchy_rank`` puts recognized reference/baseline/control terms first
  (rank 0) and recognized treated/terminal terms last (rank 2); everything else
  is rank 1 and ordered by ``natural_key``.

This is presentation only — the statistics are order-independent. When two or
more middle-rank levels are non-numeric and unrecognized, their relative order
is a pure alphabetical guess; that fallback is recorded via the ``notes``
argument so the report can declare it honestly (hidden fallbacks would obscure
that the axis/table order is heuristic, not defined by the data).
"""
import logging
import re

logger = logging.getLogger(__name__)

# Rank 0 — reference / baseline / control (sort first). Normalized (lower-case,
# whitespace and hyphens stripped) before lookup, so "Wild-Type" == "wildtype".
_REFERENCE_TERMS = {
    "baseline", "basal", "pre", "before", "t0", "day0", "time0",
    "control", "ctrl", "ctl", "con", "ctr", "nc",
    "vehicle", "veh", "untreated", "mock", "sham", "naive",
    "wt", "wildtype", "parental", "dmso",
    "unstim", "unstimulated", "uninduced", "normoxia",
    # vector / transfection controls
    "emptyvector", "empty_vector", "ev", "gfp", "scrambled", "scr",
    "sicontrol", "sictrl", "shcontrol", "shctrl",
}
# Rank 2 — treated / terminal / mutant (sort last).
_TERMINAL_TERMS = {
    "after", "post", "treated", "treatment", "trt", "tx",
    "mutant", "mut", "ko", "knockout", "ki", "knockin",
    "oe", "overexpression", "stim", "stimulated", "induced", "hypoxia",
}

_NUMERIC_LEVEL_RE = re.compile(r"\s*-?\d+(?:\.\d+)?\s*[a-zA-Z%µ°]*\s*$")


def _norm(label):
    return re.sub(r"[\s\-]+", "", str(label).strip().lower())


def _hierarchy_rank(label):
    n = _norm(label)
    if n in _REFERENCE_TERMS:
        return 0
    if n in _TERMINAL_TERMS:
        return 2
    return 1


def _natural_key(label):
    # Split into text/number chunks; wrap each in a (type, value) tuple so mixed
    # numeric/text keys stay comparable (numbers sort before text at a position).
    parts = re.split(r"(\d+(?:\.\d+)?)", str(label))
    key = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            key.append((0, float(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


def _is_numeric_level(label):
    return bool(_NUMERIC_LEVEL_RE.match(str(label)))


def natural_order(values, notes=None):
    """Return the unique ``values`` in human-friendly, row-order-independent order.

    Original values (and dtype) are preserved — only the ordering changes, so the
    result is safe to use for indexing/masking a DataFrame.

    If ``notes`` is a list and the ordering falls back to a pure alphabetical
    guess for unrecognized non-numeric levels, a one-line explanation is appended
    to it (for surfacing once in the report). Display-only callers pass no
    ``notes`` and simply ignore the fallback.
    """
    # Deterministic base: dedupe by string label, keep an original per label.
    by_label = {}
    for v in values:
        by_label.setdefault(str(v), v)

    ordered_labels = sorted(
        by_label, key=lambda s: (_hierarchy_rank(s), _natural_key(s))
    )

    ambiguous = [
        s for s in ordered_labels
        if _hierarchy_rank(s) == 1 and not _is_numeric_level(s)
    ]
    if notes is not None and len(ambiguous) >= 2:
        message = (
            "Factor levels " + ", ".join(ambiguous)
            + " were ordered alphabetically (no numeric or recognized reference "
            "pattern). Define explicit level order if a specific order is intended."
        )
        notes.append(message)
        # Log at the same point the report note is captured (callers that care
        # pass `notes`), so this fires once per analysis, not once per redraw.
        logger.warning(message)

    return [by_label[s] for s in ordered_labels]
