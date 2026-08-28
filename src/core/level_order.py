"""Human-friendly, deterministic ordering of categorical factor levels.

The order depends ONLY on the set of labels, never on DataFrame row order:

    key(label) = (rank(level_1), natural_key(level_1),
                  rank(level_2), natural_key(level_2), ...)

- ``natural_key`` is numeric-aware, so ``6h < 24h < 48h`` sort by value no matter
  what text surrounds the number (fixes lexicographic ``24h, 48h, 6h``).
- ``rank`` puts recognized reference/baseline/control terms first (rank 0) and
  recognized treated/terminal terms last (rank 2); everything else is rank 1 and
  ordered by ``natural_key``.
- A two-factor cell ("Sex=M, Geno=WT") carries one level per factor, and each is
  ranked on its own. The pair is interleaved rather than grouped, so the primary
  factor still decides the grouping and each factor's reference terms order the
  levels inside it.

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


_COMPOSITE_SEGMENT_RE = re.compile(r"^\s*[^=,]+=(?P<value>.*)$")


def _level_parts(label):
    """The level values of a label, one per factor.

    The raw-data keys for repeated-measures, mixed and two-way designs are built
    as ``f"{factor}={level}"`` (and ``"a=1, b=2"`` for a cell), so the level
    itself is buried behind a prefix that is identical for every category. Left
    in, the prefix dominates both the reference-term lookup and the sort key:
    ranking saw ``"Timepoint=Pre"`` as an unrecognized label rather than a
    baseline, and ordered ``Post`` ahead of ``Pre``.

    Split per factor rather than rejoined into one string, because the
    reference/terminal lookup is a lookup of a *level*, and a cell is not one
    level. Joining left ``"M, WT"`` to be looked up whole, which matches
    nothing, so ``Sex=F, Geno=KO`` sorted ahead of ``Sex=M, Geno=WT`` on the
    alphabet while the single-factor form put WT first. Numbers hid this --
    ``_natural_key`` finds digits anywhere in the string -- so a cell whose
    levels were T0/T1 came out right and one whose levels were WT/KO did not.
    """
    text = str(label)
    if "=" not in text:
        return [text]
    parts = []
    for segment in text.split(","):
        match = _COMPOSITE_SEGMENT_RE.match(segment)
        parts.append(match.group("value").strip() if match else segment.strip())
    return parts


def _level_part(label):
    """The level values as one string. Display and back-compatibility only."""
    return ", ".join(_level_parts(label))


def _norm(part):
    return re.sub(r"[\s\-]+", "", str(part).strip().lower())


def _rank_of(part):
    n = _norm(part)
    if n in _REFERENCE_TERMS:
        return 0
    if n in _TERMINAL_TERMS:
        return 2
    return 1


def _natkey_of(part):
    # Split into text/number chunks; wrap each in a (type, value) tuple so mixed
    # numeric/text keys stay comparable (numbers sort before text at a position).
    chunks = re.split(r"(\d+(?:\.\d+)?)", str(part))
    key = []
    for i, chunk in enumerate(chunks):
        if i % 2 == 1:
            key.append((0, float(chunk)))
        else:
            key.append((1, chunk.lower()))
    return tuple(key)


def _hierarchy_rank(label):
    """One rank per factor. A bare label still ranks as a single value."""
    return tuple(_rank_of(part) for part in _level_parts(label))


def _natural_key(label):
    key = ()
    for part in _level_parts(label):
        key += _natkey_of(part)
    return key


def _sort_key(label):
    """Rank and natural key interleaved, factor by factor.

    Not ``(all ranks, all keys)``: that would gather every control cell of every
    first-factor level together and break the grouping the chart is read by
    (all of A's bars, then all of B's). Interleaving keeps the primary factor in
    charge of the grouping and lets each factor's own reference terms order the
    levels inside it.
    """
    key = ()
    for part in _level_parts(label):
        key += ((_rank_of(part),), _natkey_of(part))
    return key


def _is_numeric_level(label):
    """True when the whole label is a number with an optional unit ("24h").

    Deliberately narrow, and NOT the test for "is this factor ordered" -- a
    number embedded in text ("Week 4") fails here while still deciding the
    order. Use :func:`order_is_defined` for that question.
    """
    return bool(_NUMERIC_LEVEL_RE.match(_level_part(label)))


def _first_difference_is_numeric(key_a, key_b):
    """Did a number decide the order of these two labels, or the alphabet?

    ``_natural_key`` splits a label into alternating text and number chunks,
    each tagged ``(0, float)`` for a number and ``(1, text)`` for text. Whatever
    chunk the two keys first disagree on is the one that placed them, so asking
    its kind answers the question exactly rather than by pattern-matching the
    label. Returns None when one key is a prefix of the other, where nothing but
    length decided.
    """
    for chunk_a, chunk_b in zip(key_a, key_b):
        if chunk_a == chunk_b:
            continue
        return chunk_a[0] == 0 and chunk_b[0] == 0
    return None


def _ambiguous_labels(ordered_labels):
    """Labels whose position rests on nothing but alphabetical order.

    A label is placed meaningfully if its rank did it (a recognized baseline or
    terminal term sits first or last by definition) or if a number separated it
    from its neighbour. Only a textual first difference between two neighbours
    of the same rank is a guess, and then both of them are guesses.

    This replaces an earlier test that asked whether each label was *entirely*
    numeric. That version called "Week 4, Week 12" and "Timepoint 1, 2, 3"
    alphabetical although their numbers had ordered them correctly, and it fired
    on every composite interaction-cell label -- which is why the resulting
    report note was muted as noise instead of the test being corrected.
    """
    ambiguous = set()
    run = []
    for label in list(ordered_labels) + [None]:
        rank = _hierarchy_rank(label) if label is not None else None
        if run and (label is None or rank != _hierarchy_rank(run[0])):
            for left, right in zip(run, run[1:]):
                decided = _first_difference_is_numeric(_natural_key(left), _natural_key(right))
                if not decided:
                    ambiguous.update((left, right))
            run = []
        if label is not None:
            run.append(label)
    return [label for label in ordered_labels if label in ambiguous]


def order_is_defined(values):
    """Report whether the level order comes from the data or from the alphabet.

    Returns ``(defined, reason)``; ``reason`` is empty when defined and
    otherwise names the levels that were merely sorted alphabetically. One
    computation with two consumers: the transparency note in the report and the
    gate that decides whether connecting individual subjects across levels would
    draw a trajectory the data never established.
    """
    by_label = {}
    for value in values:
        by_label.setdefault(str(value), value)
    ambiguous = _ambiguous_labels(
        sorted(by_label, key=_sort_key)
    )
    if not ambiguous:
        return True, ""
    return False, (
        "Factor levels " + ", ".join(ambiguous)
        + " were ordered alphabetically (no numeric or recognized reference "
        "pattern). Define explicit level order if a specific order is intended."
    )


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
        by_label, key=_sort_key
    )

    if notes is not None:
        defined, message = order_is_defined(values)
        if not defined:
            notes.append(message)
        # Log at the same point the report note is captured (callers that care
        # pass `notes`), so this fires once per analysis, not once per redraw.
        logger.warning(message)

    return [by_label[s] for s in ordered_labels]
