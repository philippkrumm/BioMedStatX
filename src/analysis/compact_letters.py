"""Compact letter display (CLD) for post-hoc results.

Groups that share a letter are not significantly different. This is the
alternative to significance brackets: brackets grow as k(k-1)/2 (15 of them at
six groups) while a letter display costs one label per group regardless of k.

The clique core below is the corrected implementation from commit ``01cd2d9``,
recovered from ``src/visualization/datavisualizer.py`` before that file was
removed in ``e9d304d``. Only the core was carried over. Its former caller,
``get_significance_letters_from_posthoc``, was deliberately left behind: it
carried a Dunnett special case that handed out letters for many-to-one designs,
which is exactly the misrepresentation :func:`letters_supported` now blocks --
two treatments significant against the control both got ``b``, asserting they do
not differ from each other when that pair was never tested. It also re-derived
significance from ``p >= alpha`` instead of honouring the flag the engine had
already set.

Deliberately dependency-free (stdlib only): the same algorithm is mirrored in
``plot_designer.js`` for the interactive figure builder, and keeping this module
readable keeps the two provably in step.
"""

import string

__all__ = ["letters_supported", "letters_from_pairs"]


def letters_supported(group_order, pairs):
    """Report whether a letter display is defensible for this post-hoc result.

    A letter display asserts something about *every* pair of groups on the plot:
    sharing a letter means "not significantly different". A pair that was never
    tested is unknown, not equal -- but the clique algorithm cannot tell the two
    apart and would silently treat it as equal. So letters are only honest when
    the comparisons cover the complete graph over the groups being drawn.

    That completeness check is deliberately structural rather than a list of
    post-hoc names. It classifies every method the project has today (Tukey,
    Games-Howell, Dunn, Nemenyi, Conover and the pairwise t-test families come
    out complete; Dunnett and the EMM/mvt many-to-one contrasts do not), it
    handles ``paired_custom`` -- where the user picks the pairs by hand in the
    comparison dialog, so no name could ever settle it -- and any test added
    later through the ``add-stat-test`` workflow is covered without touching a
    registry. A forgotten registry entry is the bug class that produced the
    IND_FDR orphan in the decision tree; here it cannot occur.

    Returns ``(supported, reason)``. ``reason`` is empty when supported and
    otherwise explains the refusal in the user's terms.
    """
    groups = [str(g) for g in group_order]
    k = len(groups)
    if k < 2:
        return False, "A letter display needs at least two groups."

    known = set(groups)
    tested = set()
    for pair in pairs or []:
        g1, g2 = str(pair.get("group1", "")), str(pair.get("group2", ""))
        if g1 in known and g2 in known and g1 != g2:
            tested.add(frozenset((g1, g2)))

    required = k * (k - 1) // 2
    if len(tested) < required:
        return False, (
            f"Letters require all {required} pairwise comparisons between the "
            f"{k} groups shown; this post-hoc provides {len(tested)}. "
            "Comparisons that were never run cannot be shown as 'not different'."
        )
    return True, ""


def letters_from_pairs(group_order, pairs, sort_by=None):
    """Assign compact letters. Call only when :func:`letters_supported` passes.

    ``pairs`` are the canonical pair dicts (``group1``, ``group2``,
    ``significant``); the ``significant`` flag is taken as given rather than
    re-derived from a p-value, so a correction applied upstream is never
    second-guessed here. ``sort_by`` maps group -> value (typically the mean) so
    that letter ``a`` tends to land on the leading group; it only affects
    labelling, never which groups share a letter.
    """
    groups = [str(g) for g in group_order]
    n = len(groups)
    if n == 0:
        return {}
    if n == 1:
        return {groups[0]: "a"}

    index_of = {g: i for i, g in enumerate(groups)}
    # not_diff[i][j] is True when i and j are NOT significantly different.
    # Untested pairs stay True, which is why letters_supported() must gate this.
    not_diff = [[True] * n for _ in range(n)]
    for pair in pairs or []:
        g1, g2 = str(pair.get("group1", "")), str(pair.get("group2", ""))
        if pair.get("significant") and g1 in index_of and g2 in index_of:
            i, j = index_of[g1], index_of[g2]
            not_diff[i][j] = not_diff[j][i] = False

    return _cld_from_not_diff(groups, not_diff, sort_by=sort_by)


def _cld_from_not_diff(groups, not_diff, sort_by=None):
    """Correct CLD via maximal cliques of the non-significance graph.

    Two groups share a letter IFF they are not significantly different --
    equivalently, they sit in a common maximal clique of mutually
    non-significant groups. This is the property the two earlier
    implementations violated: they let a *star* ({group} + all its non-different
    partners) stand in for a clique, so an intransitive pattern (A~B, B~C, but
    A#C) collapsed A, B, C onto one letter and hid the real A-C difference.

    Deterministic; a code never repeats a letter (each clique contributes one
    letter to each of its members).
    """
    n = len(groups)
    if n == 0:
        return {}
    if n == 1:
        return {groups[0]: "a"}

    adj = [set(j for j in range(n) if j != i and bool(not_diff[i][j]))
           for i in range(n)]

    # Bron-Kerbosch: enumerate every maximal clique of the non-sig graph.
    cliques = []

    def _expand(R, P, X):
        if not P and not X:
            cliques.append(set(R))
            return
        for v in list(P):
            _expand(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}

    _expand(set(), set(range(n)), set())

    # Deterministic clique order. Rank groups (higher sort_by value first, else
    # input order) and order cliques by their ranked members, so 'a' lands on
    # the leading group.
    if sort_by is not None:
        rank = {i: (-sort_by.get(groups[i], 0.0), i) for i in range(n)}
    else:
        rank = {i: (i,) for i in range(n)}
    ordered = sorted(cliques, key=lambda c: sorted(rank[i] for i in c))

    letters = {g: "" for g in groups}
    for k, clique in enumerate(ordered):
        let = (string.ascii_lowercase[k] if k < 26
               else string.ascii_lowercase[k // 26 - 1] + string.ascii_lowercase[k % 26])
        for i in sorted(clique, key=lambda i: rank[i]):
            letters[groups[i]] += let
    return letters
