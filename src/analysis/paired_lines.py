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

    if any(len(_level_parts(level)) > 1 for level in levels):
        return False, (
            "The axis here is a combination of two factors rather than one "
            "ordered sequence, so a line along it would join points that are "
            "not consecutive steps of the same path. Subject lines are drawn "
            "where the axis is a single factor."
        )

    spanning = _subjects_spanning_levels(levels, subjects)
    if not spanning:
        return False, (
            "No subject was measured at more than one level, so no line would "
            "span anything."
        )

    ordered, order_reason = order_is_defined(levels)
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
