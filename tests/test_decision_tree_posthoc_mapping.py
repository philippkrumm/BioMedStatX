"""Decision-tree node mapping for Mixed-ANOVA post-hoc tests.

Guards the fix that routes the EMM + multivariate-t treatment-vs-control test to
the between-groups node instead of the (wrong) within-subject node.
"""
from visualization.decisiontreevisualizer import DecisionTreeVisualizer as DTV


def test_emm_mvt_dunnett_maps_to_between_not_within():
    node = DTV._mixed_posthoc_node("Dunnett-type (EMM + multivariate-t, Mixed)")
    assert node == "MIXED_BETWEEN"
    assert node != "MIXED_WITHIN"


def test_tukey_maps_to_tukey():
    assert DTV._mixed_posthoc_node("Tukey HSD (Mixed)") == "MIXED_TUKEY"


def test_between_keyword_maps_to_between():
    assert DTV._mixed_posthoc_node("Between-subject pairwise") == "MIXED_BETWEEN"


def test_default_falls_to_within():
    assert DTV._mixed_posthoc_node("Paired t-tests (same subjects)") == "MIXED_WITHIN"
    assert DTV._mixed_posthoc_node("") == "MIXED_WITHIN"
