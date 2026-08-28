"""Connecting individual subjects across the levels of a paired design.

A paired test analyses within-subject differences, and a plot that shows only
the group distributions hides exactly that. Two datasets can produce identical
boxes while one has every subject moving the same way and the other has half
moving each way -- the same picture, opposite results. Lines put the analysed
quantity back on the plot.

They are not always right, and the conditions are structural rather than a
matter of taste:

* **Subject identity.** Without it there is nothing to connect. Independent
  designs never qualify.
* **A defined level order.** A line asserts a path from one level to the next.
  Where the order is an alphabetical guess (``Drug A`` before ``Drug B``) the
  path is an artefact of the labels, so the question is delegated to
  :func:`core.level_order.order_is_defined` -- the same computation whose note
  in the report explains the refusal.
* **Levels a subject can actually move between.** A mixed design's axis is a
  grid of cells, and a line across the whole of it would be nonsense -- but a
  subject never changes its between group, so no line can cross one. Each
  between group's cells therefore form a block, the lines run inside a block
  along the within factor, and the structure is exactly the repeated-measures
  case drawn once per block. The blocks are read off the subjects rather than
  the labels, so nothing depends on which factor was written first, and the
  ordering question is asked of the part that varies inside a block: whether
  ``Site=Aachen`` precedes ``Site=Bonn`` says nothing about the path a line
  asserts.
* **A readable number of subjects.** Past a few dozen the lines stop being
  readable and become a wash. Above the limit the plot is left as it is; the
  right answer there is a difference-based display, which is its own job.

Dependency-free by design: the interactive figure builder mirrors this logic in
JavaScript, and keeping the Python side small keeps the two comparable.
"""

from core.level_order import _level_parts, order_is_defined

# Weissgerber et al. (2015) put the practical ceiling for readable spaghetti
# overlays at roughly this many series. Nothing about the number is sacred --
# it is a legibility threshold, not a statistical one -- so it lives here as a
# single named value rather than sprinkled through the renderers.
PAIRED_LINE_MAX_SUBJECTS = 30

__all__ = ["PAIRED_LINE_MAX_SUBJECTS", "paired_lines_supported", "build_paired_trajectories"]


def _subjects_spanning_levels(group_order, subjects):
    """Subjects that appear in at least two of the levels being drawn."""
    seen = {}
    for group in group_order:
        for subject in subjects.get(group) or []:
            seen.setdefault(str(subject), set()).add(str(group))
    return {subject for subject, groups in seen.items() if len(groups) >= 2}


def _blocks(levels, subjects):
    """The levels split into the sets a single subject can move within.

    Derived from the subjects, not from the label: two levels belong together
    exactly when some subject appears in both. In a purely within-subject design
    that is one block containing everything. In a mixed design a subject belongs
    to one between group for the whole study, so each between group's cells form
    their own block and no line can leave it -- which is the reason lines are
    defensible here at all.

    Reading it off the data rather than the label also means nothing depends on
    which factor was written first, and a design with an unexpected structure
    produces blocks that describe what it actually is.
    """
    parent = {level: level for level in levels}

    def find(level):
        while parent[level] != level:
            parent[level] = parent[parent[level]]
            level = parent[level]
        return level

    by_subject = {}
    for level in levels:
        for subject in subjects.get(level) or []:
            by_subject.setdefault(str(subject), []).append(level)
    for seen in by_subject.values():
        for other in seen[1:]:
            root_a, root_b = find(seen[0]), find(other)
            if root_a != root_b:
                parent[root_a] = root_b

    grouped = {}
    for level in levels:
        grouped.setdefault(find(level), []).append(level)
    return [grouped[root] for root in dict.fromkeys(find(level) for level in levels)]


def _varying_part(block):
    """What actually changes from one level of a block to the next.

    Inside a block the between levels are constant by construction, so only the
    within factor moves. Ordering has to be judged on that part alone: whether
    ``Site=Aachen`` precedes ``Site=Bonn`` is a question about the axis, not
    about the path a line asserts, and letting it decide would refuse lines for
    a reason that has nothing to do with them.
    """
    parts = [_level_parts(level) for level in block]
    if len({len(part) for part in parts}) != 1:
        return list(block)
    varying = [index for index in range(len(parts[0]))
               if len({part[index] for part in parts}) > 1]
    if not varying:
        return list(block)
    return [", ".join(part[index] for index in varying) for part in parts]


def _is_contiguous(block, levels):
    """Are this block's levels drawn side by side?

    A trajectory is plotted at the level's position on the axis, so a block
    split by another block's bars would be drawn as a line reaching across
    them. The cell ordering keeps each block together; this is the check that
    says so rather than assuming it.
    """
    positions = sorted(levels.index(level) for level in block)
    return positions == list(range(positions[0], positions[0] + len(positions)))


def paired_lines_supported(group_order, subjects, max_subjects=PAIRED_LINE_MAX_SUBJECTS):
    """Report whether subject lines are defensible here.

    Returns ``(supported, reason)``; ``reason`` is empty when supported and
    otherwise says which condition failed, in the reader's terms.
    """
    levels = [str(g) for g in group_order or []]
    if len(levels) < 2:
        return False, "Subject lines need at least two levels to connect."

    if not isinstance(subjects, dict) or not subjects:
        return False, (
            "No subject identity in this result, so there is nothing to connect. "
            "Independent designs measure different individuals per group."
        )

    spanning = _subjects_spanning_levels(levels, subjects)
    if not spanning:
        return False, (
            "No subject was measured at more than one level, so no line would "
            "span anything."
        )

    # Lines run inside a block and never between blocks. In a single-factor
    # design there is one block and this is the ordering rule as it always was;
    # in a mixed design each between group is its own block, and the ordering
    # question is asked of the within factor, which is the only thing a line
    # moves along.
    for block in _blocks(levels, subjects):
        if len(block) < 2:
            continue
        if not _is_contiguous(block, levels):
            return False, (
                "The levels a subject was measured at are not drawn next to "
                "each other, so a line would have to reach across other groups "
                "to join them."
            )
        ordered, order_reason = order_is_defined(_varying_part(block))
        if not ordered:
            return False, (
                "A line between levels asserts a path, and this order is "
                "alphabetical rather than given by the data. " + order_reason
            )

    if len(spanning) > max_subjects:
        return False, (
            f"{len(spanning)} subjects exceed the {max_subjects} that stay "
            "readable as individual lines; the plot is left without them."
        )

    return True, ""


def build_paired_trajectories(group_order, raw_data, subjects):
    """One entry per subject: its value at each level, in level order.

    Values are matched to levels through the subject label, never through
    position -- the raw values are stored per level in whatever order the frame
    held, so the k-th value of two levels can belong to different people. A
    subject measured at only one level is dropped rather than drawn as a point
    pretending to be a line.
    """
    levels = [str(g) for g in group_order or []]
    if not isinstance(raw_data, dict) or not isinstance(subjects, dict):
        return []

    points_by_subject = {}
    for index, group in enumerate(levels):
        values = raw_data.get(group) or []
        labels = subjects.get(group) or []
        for position, label in enumerate(labels):
            if position >= len(values):
                continue
            try:
                value = float(values[position])
            except (TypeError, ValueError):
                continue
            # A subject listed twice within one level has no single position on
            # the axis; keep the first and leave the duplicate out rather than
            # drawing a line that doubles back.
            points_by_subject.setdefault(str(label), {}).setdefault(index, value)

    trajectories = []
    for subject in sorted(points_by_subject):
        points = points_by_subject[subject]
        if len(points) < 2:
            continue
        trajectories.append({
            "subject": subject,
            "points": [{"level_index": i, "group": levels[i], "value": points[i]}
                       for i in sorted(points)],
        })
    return trajectories
