"""Guard: the decision-tree post-hoc leaves must match the post-hoc options the
dialogs actually offer per design, so the tree can't silently drift from the code
(as RM_TUKEY / the one-way-vs-two-way conflation did before the pre-2.0 fixes).

Dialog options (stats_functions.select_posthoc_test_dialog):
  one-way (Welch): games_howell, dunnett, paired_custom
  two-way:         tukey, paired_custom
  RM:              paired_custom, emm_mvt
  mixed:           paired_custom, emm_mvt   (mixed leaves are revised in feature B)
"""
from visualization.decisiontreevisualizer import DecisionTreeVisualizer as DTV


def _tree():
    tree = DTV.get_tree_json({"model_type": "", "test": "one_way_anova", "p_value": 0.01})
    edges = {(e["source"], e["target"]) for e in tree["edges"]}
    node_ids = {n["id"] for n in tree["nodes"]}
    return edges, node_ids


def test_one_way_posthoc_offers_games_howell_dunnett_holm_not_tukey():
    edges, node_ids = _tree()
    assert ("IND_OW_POSTHOC", "IND_GAMES_HOWELL") in edges
    assert ("IND_OW_POSTHOC", "IND_DUNNETT") in edges
    assert ("IND_OW_POSTHOC", "IND_OW_HOLM") in edges
    # Welch one-way never offers a plain Tukey
    assert ("IND_OW_POSTHOC", "IND_TUKEY") not in edges


def test_two_way_posthoc_offers_tukey_holm_not_dunnett_or_games_howell():
    edges, node_ids = _tree()
    assert ("IND_TW_POSTHOC", "IND_TUKEY") in edges
    assert ("IND_TW_POSTHOC", "IND_TW_HOLM") in edges
    assert ("IND_TW_POSTHOC", "IND_DUNNETT") not in edges
    assert ("IND_TW_POSTHOC", "IND_GAMES_HOWELL") not in edges


def test_rm_posthoc_is_emm_and_paired_no_tukey():
    edges, node_ids = _tree()
    assert ("RM_POSTHOC", "RM_EMM") in edges
    assert ("RM_POSTHOC", "RM_PAIRED_TESTS") in edges
    assert "RM_TUKEY" not in node_ids


def test_removed_stale_nodes_are_gone():
    edges, node_ids = _tree()
    for gone in ("IND_POSTHOC", "IND_HOLM_SIDAK", "RM_TUKEY", "MIXED_TUKEY"):
        assert gone not in node_ids, gone


def test_mixed_posthoc_is_interaction_first_with_named_tests():
    """Feature B: the mixed post-hoc branches on the interaction first, then into
    simple main effects (YES) or main-effects-only (NO), each leaf naming the
    actual test the app runs (#5)."""
    tree = DTV.get_tree_json({"model_type": "", "test": "one_way_anova", "p_value": 0.01})
    edges = {(e["source"], e["target"]) for e in tree["edges"]}
    labels = {n["id"]: n["label"] for n in tree["nodes"]}

    # interaction question -> two branches
    assert ("MIXED_POSTHOC", "MIXED_SME") in edges
    assert ("MIXED_POSTHOC", "MIXED_ME") in edges
    # YES -> simple main effects
    assert ("MIXED_SME", "MIXED_BETWEEN") in edges
    assert ("MIXED_SME", "MIXED_WITHIN") in edges
    # NO -> main effects only
    assert ("MIXED_ME", "MIXED_MARG_BETWEEN") in edges
    assert ("MIXED_ME", "MIXED_MARG_WITHIN") in edges

    assert "interaction" in labels["MIXED_POSTHOC"].lower()
    assert "Paired t" in labels["MIXED_WITHIN"]
    assert "Indep" in labels["MIXED_BETWEEN"]
    assert "Main Effect of Group" in labels["MIXED_MARG_BETWEEN"]
    assert "Main Effect of Time" in labels["MIXED_MARG_WITHIN"]
